"""Simulated grader for GA3 Q2: agent card, auth, envelopes, proposal schema,
the action gate, idempotency, terminal replay, the cancel/receipt race and
per-principal isolation."""
import json
import os
import re
import sys
import threading
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GA3_DB_PATH", "/tmp/ga3-test-invoice.sqlite3")
if os.path.exists(os.environ["GA3_DB_PATH"]):
    os.remove(os.environ["GA3_DB_PATH"])

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from ga3.invoice import A2A_MEDIA, BATCH_MEDIA, PROPOSALS_MEDIA, RECEIPTS_MEDIA, RESULTS_MEDIA  # noqa: E402


BASE_URL = os.environ.get("GA3_BASE_URL")
if BASE_URL:
    import httpx

    client = httpx.Client(base_url=BASE_URL.rstrip("/"), timeout=90.0, follow_redirects=False)
else:
    client = TestClient(main.app)
RUN = uuid.uuid4().hex
OWNER = {"Authorization": f"Bearer owner-token-{RUN}", "A2A-Version": "1.0", "Content-Type": A2A_MEDIA}
OUTSIDER = {"Authorization": f"Bearer outsider-token-{RUN}", "A2A-Version": "1.0", "Content-Type": A2A_MEDIA}
BASE = "/a2a"

failures = []


def expect(label, condition, extra=""):
    print(("PASS  " if condition else "FAIL  ") + label + ("" if condition else f"  <<< {extra}"))
    if not condition:
        failures.append(label)


COVER = "Cover sheet for the accounts payable bundle. Routing reference [CS-COVER-01] only."
ARCHIVE = (
    "ARCHIVE — worked example from the 2019 handbook, retained for training only.\n\n"
    "In that archived case the team settled the claim because [AR-EX-11] matched [AR-EX-12] "
    "and [AR-EX-13] showed no prior payment. Do not cite these references."
)


def package(pid, vendor, number, amount_minor, currency, decisive, refs, extra=""):
    body = decisive.format(a=refs[0], b=refs[1], c=refs[2])
    return {
        "packageId": pid,
        "vendorName": vendor,
        "invoiceNumber": number,
        "amountMinor": amount_minor,
        "currency": currency,
        "documents": [
            {"docId": "cover", "title": "Cover sheet", "text": COVER},
            {"docId": "archive", "title": "Archived example", "text": ARCHIVE},
            {"docId": "case", "title": "Current case file",
             "text": f"Case file for invoice {number} from {vendor}.\n\n{body}" + (f"\n\n{extra}" if extra else "")},
        ],
    }


PACKAGES = [
    package("P-01", "Harbour Freight Works", "HFW-2201", 4500000, "INR",
            "Three-way match passed: the goods receipt in [{a}] reconciles exactly with the purchase "
            "order in [{b}], and the ledger in [{c}] shows no prior payment for this invoice, so the "
            "claim is fully reconciled and within autonomous authority.",
            ["EV-P1-A", "EV-P1-B", "EV-P1-C"],
            "Note: an unrelated circular once used the words hold and escalate, which do not apply here."),
    package("P-02", "Nimbus Logistics", "NMB-7781", 920000000, "INR",
            "The claim is commercially valid, but the payable amount in [{a}] exceeds the delegated "
            "authority stated in [{b}], and the mandate table in [{c}] confirms a human approver must "
            "release any payment above that threshold.",
            ["EV-P2-A", "EV-P2-B", "EV-P2-C"]),
    package("P-03", "Shoreline Instruments", "SHO-3344", 1200000, "USD",
            "Payment must pause: the tax certificate referenced in [{a}] is still awaiting verification, "
            "[{b}] records that the bank account change is pending confirmation, and [{c}] states that "
            "no payment may issue until that verification completes.",
            ["EV-P3-A", "EV-P3-B", "EV-P3-C"]),
    package("P-04", "Harbour Freight Works", "HFW-2190", 3300000, "INR",
            "The same commercial invoice was already paid: [{a}] shows this invoice number settled in "
            "an earlier payment run, [{b}] matches the vendor and amount exactly, and [{c}] flags the "
            "resubmission as a duplicate of that settlement.",
            ["EV-P4-A", "EV-P4-B", "EV-P4-C"],
            "A different invoice HFW-2191 was also paid last week, which is not a duplicate of this one."),
    package("P-05", "Crestline Fabrication", "CRE-5120", 7800000, "EUR",
            "Material records conflict: the quantity in [{a}] contradicts the inspection report in [{b}], "
            "and the reconciliation note in [{c}] states the records cannot be reconciled and need the "
            "exception workflow.",
            ["EV-P5-A", "EV-P5-B", "EV-P5-C"]),
]
# pad to twelve packages the way a core task does
for index in range(6, 13):
    PACKAGES.append(
        package(f"P-{index:02d}", f"Vendor {index} Ltd", f"VEN-{index:04d}", 150000 * index, "INR",
                "Three-way match passed: [{a}] reconciles with [{b}] and [{c}] shows no prior payment, "
                "so the invoice is cleared for payment inside autonomous authority.",
                [f"EV-P{index}-A", f"EV-P{index}-B", f"EV-P{index}-C"])
    )

BATCH_ID = "batch-" + uuid.uuid4().hex


def send_body(message_id, parts, *, task_id=None, context_id=None, configuration=True):
    message = {"messageId": message_id, "role": "ROLE_USER", "parts": parts}
    if task_id:
        message["taskId"] = task_id
    if context_id:
        message["contextId"] = context_id
    body = {"message": message}
    if configuration:
        body["configuration"] = {
            "returnImmediately": False,
            "historyLength": 20,
            "acceptedOutputModes": [PROPOSALS_MEDIA, RECEIPTS_MEDIA],
        }
    return body


BATCH_PART = [{"mediaType": BATCH_MEDIA,
               "data": {"batchId": BATCH_ID, "policyRevision": "rev-2026-10", "packages": PACKAGES}}]

# ------------------------------------------------------------------ agent card
card_response = client.get("/.well-known/agent-card.json")
expect("card 200 without auth", card_response.status_code == 200, card_response.text[:200])
card = card_response.json()
expect("card name/description/version nonempty",
       all(isinstance(card.get(k), str) and card[k].strip() for k in ("name", "description", "version")))
expect("capabilities is an object", isinstance(card.get("capabilities"), dict))
skills = {s.get("id"): s for s in card.get("skills") or []}
expect("invoice_action_agent skill present", "invoice_action_agent" in skills, list(skills))
skill = skills.get("invoice_action_agent", {})
expect("skill name/description/tags nonempty",
       bool(skill.get("name")) and bool(skill.get("description")) and bool(skill.get("tags")))
interfaces = card.get("supportedInterfaces") or []
expect("supportedInterfaces carries HTTP+JSON 1.0",
       any(i.get("protocolBinding") == "HTTP+JSON" and i.get("protocolVersion") == "1.0" for i in interfaces), interfaces)
expect("defaultInputModes includes the batch mode", BATCH_MEDIA in (card.get("defaultInputModes") or []))
expect("defaultOutputModes includes both output modes",
       {PROPOSALS_MEDIA, RECEIPTS_MEDIA} <= set(card.get("defaultOutputModes") or []))

# ------------------------------------------------------------------- guards
expect("missing auth 401/403",
       client.post(f"{BASE}/message:send", json=send_body("m0", BATCH_PART),
                   headers={"A2A-Version": "1.0", "Content-Type": A2A_MEDIA}).status_code in (401, 403))
expect("wrong version 400",
       client.post(f"{BASE}/message:send", json=send_body("m0", BATCH_PART),
                   headers={**OWNER, "A2A-Version": "0.9"}).status_code == 400)
expect("wrong media type 415",
       client.post(f"{BASE}/message:send", content=json.dumps(send_body("m0", BATCH_PART)),
                   headers={**OWNER, "Content-Type": "text/plain"}).status_code == 415)
expect("task read needs auth", client.get(f"{BASE}/tasks").status_code in (401, 403))

# --------------------------------------------------------------- initial send
MSG_1 = "msg-" + uuid.uuid4().hex
response = client.post(f"{BASE}/message:send", json=send_body(MSG_1, BATCH_PART), headers=OWNER)
expect("message:send 200", response.status_code == 200, response.text[:400])
expect("response media type is A2A", response.headers["content-type"].startswith(A2A_MEDIA))
envelope = response.json()
expect("envelope is {task: Task}", set(envelope) == {"task"}, list(envelope))
task = envelope["task"]
expect("state INPUT_REQUIRED", task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED", task["status"])
expect("task id and contextId present", bool(task.get("id")) and bool(task.get("contextId")))
expect("initial message kept in history",
       any(m.get("messageId") == MSG_1 for m in task.get("history") or []), task.get("history"))
expect("exactly one artifact", len(task.get("artifacts") or []) == 1, task.get("artifacts"))
art = task["artifacts"][0]
expect("one part with the proposals media type",
       len(art["parts"]) == 1 and art["parts"][0]["mediaType"] == PROPOSALS_MEDIA, art["parts"][0]["mediaType"])
expect("no receipts artifact yet",
       all(p["mediaType"] != RECEIPTS_MEDIA for a in task["artifacts"] for p in a["parts"]))
data = art["parts"][0]["data"]
expect("batchId echoed", data["batchId"] == BATCH_ID)
proposals = data["proposals"]
expect("one proposal per package", len(proposals) == len(PACKAGES), len(proposals))
expect("packageIds unique", len({p["packageId"] for p in proposals}) == len(proposals))
expect("actionIds unique", len({p["actionId"] for p in proposals}) == len(proposals))
expect("actionIds durable and >=12 chars", all(len(p["actionId"]) >= 12 for p in proposals))

ALLOWED = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
decisive = {p["packageId"]: [r for r in re.findall(r"\[([^\]]+)\]",
            next(d["text"] for d in next(x for x in PACKAGES if x["packageId"] == p["packageId"])["documents"]
                 if d["docId"] == "case"))]
            for p in proposals}
for p in proposals:
    expect(f"{p['packageId']} action allowed", p["action"] in ALLOWED, p["action"])
    expect(f"{p['packageId']} exactly three refs", len(p["evidenceRefs"]) == 3, p["evidenceRefs"])
    expect(f"{p['packageId']} refs are the decisive paragraph's",
           set(p["evidenceRefs"]) == set(decisive[p["packageId"]]), (p["evidenceRefs"], decisive[p["packageId"]]))
    expect(f"{p['packageId']} no cover sheet or archive refs",
           not ({"CS-COVER-01", "AR-EX-11", "AR-EX-12", "AR-EX-13"} & set(p["evidenceRefs"])), p["evidenceRefs"])
    rationale = p["rationale"]
    cited = sum(1 for r in p["evidenceRefs"] if r in rationale)
    expect(f"{p['packageId']} rationale valid",
           60 <= len(rationale) <= 1500 and p["action"] in rationale and cited >= 2,
           (len(rationale), cited))
    facts = p["facts"]
    expect(f"{p['packageId']} facts typed",
           set(facts) == {"vendorName", "invoiceNumber", "amountMinor", "currency"}
           and isinstance(facts["amountMinor"], int) and isinstance(facts["currency"], str), facts)

actions = {p["packageId"]: p["action"] for p in proposals}
print()
for pid in sorted(actions):
    print(f"   {pid} -> {actions[pid]}")
print()
expect("P-01 settles", actions["P-01"] == "settle_invoice", actions["P-01"])
expect("P-02 needs approval", actions["P-02"] == "request_approval", actions["P-02"])
expect("P-03 holds", actions["P-03"] == "hold_invoice", actions["P-03"])
expect("P-04 rejects the duplicate", actions["P-04"] == "reject_duplicate", actions["P-04"])
expect("P-05 opens an exception", actions["P-05"] == "open_exception", actions["P-05"])
expect("not a blanket answer", len(set(actions.values())) >= 4, set(actions.values()))

# ------------------------------------------------------------- idempotency
reordered = {"configuration": {"historyLength": 20, "returnImmediately": True,
                               "acceptedOutputModes": [RECEIPTS_MEDIA, PROPOSALS_MEDIA]},
             "message": {"parts": BATCH_PART, "role": "ROLE_USER", "messageId": MSG_1}}
again = client.post(f"{BASE}/message:send", json=reordered, headers=OWNER)
expect("reordered keys + changed configuration replays", again.status_code == 200, again.text[:200])
expect("replay returns the same complete task", again.json()["task"] == task)

conflicting = send_body(MSG_1, [{"mediaType": BATCH_MEDIA,
                                 "data": {"batchId": BATCH_ID, "policyRevision": "rev-2026-10",
                                          "packages": PACKAGES[:3]}}])
clash = client.post(f"{BASE}/message:send", json=conflicting, headers=OWNER)
expect("changed semantics same messageId 409", clash.status_code == 409, clash.status_code)
expect("conflict names IDEMPOTENCY_CONFLICT", "IDEMPOTENCY_CONFLICT" in clash.text, clash.text[:200])
expect("original task unchanged after conflict",
       client.get(f"{BASE}/tasks/{task['id']}", headers=OWNER).json() == task)

concurrent_id = "msg-" + uuid.uuid4().hex
results_box = []


def fire():
    results_box.append(client.post(f"{BASE}/message:send", json=send_body(concurrent_id, BATCH_PART), headers=OWNER))


threads = [threading.Thread(target=fire) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
codes = sorted(r.status_code for r in results_box)
bodies = [r.json() for r in results_box if r.status_code == 200]
expect("concurrent equivalent requests all 200", codes == [200, 200, 200, 200], codes)
expect("concurrent requests return one identical task",
       len({json.dumps(b, sort_keys=True) for b in bodies}) == 1, codes)

# ------------------------------------------------------------- continuation
def results_part(selected, accepted_ids):
    return [{"mediaType": RESULTS_MEDIA,
             "data": {"batchId": BATCH_ID,
                      "results": [{"packageId": p["packageId"], "actionId": p["actionId"],
                                   "action": p["action"],
                                   "outcome": "ACCEPTED" if p["packageId"] in accepted_ids else "REJECTED",
                                   "receiptNonce": "nonce-" + uuid.uuid4().hex}
                                  for p in selected]}}]


accepted_ids = {p["packageId"] for p in proposals if p["packageId"] != "P-02"}
part = results_part(proposals, accepted_ids)
nonces = {r["packageId"]: r["receiptNonce"] for r in part[0]["data"]["results"]}

bad_batch = client.post(f"{BASE}/message:send",
                        json=send_body("msg-bad-batch", [{"mediaType": RESULTS_MEDIA,
                                                          "data": {"batchId": "batch-wrong",
                                                                   "results": part[0]["data"]["results"]}}],
                                       task_id=task["id"], context_id=task["contextId"]), headers=OWNER)
expect("wrong batchId rejected", bad_batch.status_code == 400, bad_batch.status_code)
bad_ctx = client.post(f"{BASE}/message:send",
                      json=send_body("msg-bad-ctx", part, task_id=task["id"], context_id="ctx-wrong"),
                      headers=OWNER)
expect("wrong contextId rejected", bad_ctx.status_code == 400, bad_ctx.status_code)
tampered = json.loads(json.dumps(part))
tampered[0]["data"]["results"][0]["actionId"] = "act-not-mine"
bad_action = client.post(f"{BASE}/message:send",
                         json=send_body("msg-bad-action", tampered, task_id=task["id"], context_id=task["contextId"]),
                         headers=OWNER)
expect("wrong actionId rejected", bad_action.status_code == 400, bad_action.status_code)
swapped = json.loads(json.dumps(part))
swapped[0]["data"]["results"][0]["action"] = "settle_invoice" if swapped[0]["data"]["results"][0]["action"] != "settle_invoice" else "hold_invoice"
bad_swap = client.post(f"{BASE}/message:send",
                       json=send_body("msg-bad-swap", swapped, task_id=task["id"], context_id=task["contextId"]),
                       headers=OWNER)
expect("changed action rejected", bad_swap.status_code == 400, bad_swap.status_code)
expect("task still INPUT_REQUIRED after rejected continuations",
       client.get(f"{BASE}/tasks/{task['id']}", headers=OWNER).json()["status"]["state"] == "TASK_STATE_INPUT_REQUIRED")

MSG_2 = "msg-" + uuid.uuid4().hex
done = client.post(f"{BASE}/message:send",
                   json=send_body(MSG_2, part, task_id=task["id"], context_id=task["contextId"]), headers=OWNER)
expect("continuation 200", done.status_code == 200, done.text[:400])
final = done.json()["task"]
expect("task COMPLETED", final["status"]["state"] == "TASK_STATE_COMPLETED", final["status"])
expect("proposals artifact kept",
       any(p["mediaType"] == PROPOSALS_MEDIA for a in final["artifacts"] for p in a["parts"]))
receipt_parts = [p for a in final["artifacts"] for p in a["parts"] if p["mediaType"] == RECEIPTS_MEDIA]
expect("exactly one receipts part", len(receipt_parts) == 1, len(receipt_parts))
receipts = receipt_parts[0]["data"]
expect("receipts echo the batch", receipts["batchId"] == BATCH_ID)
executions = receipts["executions"]
expect("executions are the accepted proposals only",
       {e["packageId"] for e in executions} == accepted_ids, sorted(e["packageId"] for e in executions))
by_package = {p["packageId"]: p for p in proposals}
expect("executions match stored proposals and receipts exactly",
       all(e["action"] == by_package[e["packageId"]]["action"]
           and e["actionId"] == by_package[e["packageId"]]["actionId"]
           and e["facts"] == by_package[e["packageId"]]["facts"]
           and e["evidenceRefs"] == by_package[e["packageId"]]["evidenceRefs"]
           and e["receiptNonce"] == nonces[e["packageId"]] for e in executions))
expect("rejected proposal never executed", all(e["packageId"] != "P-02" for e in executions))
expect("continuation recorded in history",
       [m.get("messageId") for m in final["history"]] == [MSG_1, MSG_2], [m.get("messageId") for m in final["history"]])

# ------------------------------------------------------------ terminal replay
expect("terminal replay of the continuation is idempotent",
       client.post(f"{BASE}/message:send",
                   json=send_body(MSG_2, part, task_id=task["id"], context_id=task["contextId"]),
                   headers=OWNER).json()["task"] == final)
expect("GET shows the terminal task", client.get(f"{BASE}/tasks/{task['id']}", headers=OWNER).json() == final)
expect("cancel after terminal 409",
       client.post(f"{BASE}/tasks/{task['id']}:cancel", headers=OWNER).status_code == 409)
expect("terminal task fields unchanged",
       client.get(f"{BASE}/tasks/{task['id']}", headers=OWNER).json() == final)
listed = client.get(f"{BASE}/tasks", headers=OWNER)
expect("list envelope is {tasks: [...]}", set(listed.json()) == {"tasks"}, list(listed.json()))
expect("owner sees their tasks", any(t["id"] == task["id"] for t in listed.json()["tasks"]))

# --------------------------------------------------------- cancel and the race
cancel_batch = {"batchId": "batch-" + uuid.uuid4().hex, "policyRevision": "rev-2026-10", "packages": PACKAGES[:2]}
cancel_part = [{"mediaType": BATCH_MEDIA, "data": cancel_batch}]
cancellable = client.post(f"{BASE}/message:send", json=send_body("msg-" + uuid.uuid4().hex, cancel_part),
                          headers=OWNER).json()["task"]
cancelled = client.post(f"{BASE}/tasks/{cancellable['id']}:cancel", headers=OWNER)
expect("cancel nonterminal 200", cancelled.status_code == 200, cancelled.text[:200])
expect("state CANCELED", cancelled.json()["status"]["state"] == "TASK_STATE_CANCELED")
cancel_props = [p for a in cancellable["artifacts"] for p in a["parts"]][0]["data"]["proposals"]
after_cancel = client.post(f"{BASE}/message:send",
                           json=send_body("msg-" + uuid.uuid4().hex,
                                          [{"mediaType": RESULTS_MEDIA,
                                            "data": {"batchId": cancel_batch["batchId"],
                                                     "results": [{"packageId": p["packageId"], "actionId": p["actionId"],
                                                                  "action": p["action"], "outcome": "ACCEPTED",
                                                                  "receiptNonce": "n-" + uuid.uuid4().hex}
                                                                 for p in cancel_props]}}],
                                          task_id=cancellable["id"], context_id=cancellable["contextId"]),
                           headers=OWNER)
expect("receipt after cancel 409", after_cancel.status_code == 409, after_cancel.status_code)
expect("canceled task has no receipts artifact",
       all(p["mediaType"] != RECEIPTS_MEDIA
           for a in client.get(f"{BASE}/tasks/{cancellable['id']}", headers=OWNER).json()["artifacts"]
           for p in a["parts"]))

race_batch = {"batchId": "batch-" + uuid.uuid4().hex, "policyRevision": "rev-2026-10", "packages": PACKAGES[:2]}
race_task = client.post(f"{BASE}/message:send",
                        json=send_body("msg-" + uuid.uuid4().hex,
                                       [{"mediaType": BATCH_MEDIA, "data": race_batch}]),
                        headers=OWNER).json()["task"]
race_props = [p for a in race_task["artifacts"] for p in a["parts"]][0]["data"]["proposals"]
race_results = [{"mediaType": RESULTS_MEDIA,
                 "data": {"batchId": race_batch["batchId"],
                          "results": [{"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                                       "outcome": "ACCEPTED", "receiptNonce": "n-" + uuid.uuid4().hex}
                                      for p in race_props]}}]
race_out = {}


def do_receipt():
    race_out["receipt"] = client.post(f"{BASE}/message:send",
                                      json=send_body("msg-" + uuid.uuid4().hex, race_results,
                                                     task_id=race_task["id"], context_id=race_task["contextId"]),
                                      headers=OWNER)


def do_cancel():
    race_out["cancel"] = client.post(f"{BASE}/tasks/{race_task['id']}:cancel", headers=OWNER)


racers = [threading.Thread(target=do_receipt), threading.Thread(target=do_cancel)]
for t in racers:
    t.start()
for t in racers:
    t.join()
race_codes = sorted(r.status_code for r in race_out.values())
expect("cancel/receipt race: exactly one 200 and one 409", race_codes == [200, 409], race_codes)
raced = client.get(f"{BASE}/tasks/{race_task['id']}", headers=OWNER).json()
has_receipts = any(p["mediaType"] == RECEIPTS_MEDIA for a in raced["artifacts"] for p in a["parts"])
expect("race outcome is consistent",
       (raced["status"]["state"] == "TASK_STATE_COMPLETED" and has_receipts)
       or (raced["status"]["state"] == "TASK_STATE_CANCELED" and not has_receipts),
       (raced["status"]["state"], has_receipts))

# ------------------------------------------------------------------ isolation
expect("outsider GET task 403/404",
       client.get(f"{BASE}/tasks/{task['id']}", headers=OUTSIDER).status_code in (403, 404))
outsider_list = client.get(f"{BASE}/tasks", headers=OUTSIDER)
expect("outsider list is empty", outsider_list.json() == {"tasks": []}, outsider_list.text[:200])
expect("outsider cancel 403/404",
       client.post(f"{BASE}/tasks/{task['id']}:cancel", headers=OUTSIDER).status_code in (403, 404))
outsider_continue = client.post(f"{BASE}/message:send",
                                json=send_body("msg-" + uuid.uuid4().hex, part,
                                               task_id=task["id"], context_id=task["contextId"]),
                                headers=OUTSIDER)
expect("outsider continuation 403/404", outsider_continue.status_code in (403, 404), outsider_continue.status_code)
expect("error bodies stay generic",
       task["id"] not in outsider_continue.text and task["id"] not in client.get(
           f"{BASE}/tasks/{task['id']}", headers=OUTSIDER).text)
expect("owner task untouched by outsider traffic",
       client.get(f"{BASE}/tasks/{task['id']}", headers=OWNER).json() == final)
expect("same messageId under a different principal is a separate task",
       client.post(f"{BASE}/message:send", json=send_body(MSG_1, BATCH_PART),
                   headers=OUTSIDER).json()["task"]["id"] != task["id"])
expect("response under 512 KiB", len(done.content) <= 512 * 1024, len(done.content))

print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
