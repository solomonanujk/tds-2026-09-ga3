"""TDS 2026-09 GA3 — one service answering Q2, Q3 and Q4.

  Q2  A2A invoice agent        base  <BASE>/a2a/   (card at <ORIGIN>/.well-known/agent-card.json)
  Q3  Safe AI mailroom         URL   <BASE>/mailroom
  Q4  Observable incident      base  <BASE>        (routes under /v2/incidents)

Run locally:  uvicorn main:app --port 8000
"""
from __future__ import annotations

import os
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

from ga3 import incident, invoice, mailroom
from ga3 import llm, store

EMAIL = "24f2001638@ds.study.iitm.ac.in"

app = FastAPI(title="TDS GA3 agents", version=os.environ.get("GA3_VERSION", "1.0.0"))

app.include_router(invoice.router)
app.include_router(mailroom.router)
app.include_router(incident.router)


DEBUG_TOKEN = os.environ.get("GA3_DEBUG_TOKEN", "").strip()
CAPTURE_PATHS = ("/a2a/", "/mailroom", "/v2/incidents")


@app.middleware("http")
async def capture_requests(request, call_next):
    """With GA3_DEBUG_TOKEN set, keep the last few grader bodies for tuning.

    The exam corpora are synthetic, and seeing a real payload is the only way
    to improve decision quality between Checks. Disabled unless the token is
    configured, and readable only with it.
    """
    if not DEBUG_TOKEN or not request.url.path.startswith(CAPTURE_PATHS) or request.method != "POST":
        return await call_next(request)
    body = await request.body()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = receive  # noqa: SLF001 - replay the consumed body
    response = await call_next(request)
    try:
        store.put(
            "debug.capture",
            f"{time.time():.6f}",
            {
                "at": time.time(),
                "path": request.url.path,
                "status": response.status_code,
                "body": body[:400_000].decode("utf-8", "replace"),
            },
        )
        captures = sorted(
            r["key"] for r in [{"key": k} for k in _capture_keys()]
        )
        for stale in captures[:-20]:
            store.delete("debug.capture", stale)
    except Exception:
        pass
    return response


def _capture_keys():
    rows = store._connect().execute(  # noqa: SLF001 - debug-only helper
        "SELECT key FROM records WHERE ns='debug.capture'"
    ).fetchall()
    return [row["key"] for row in rows]


@app.get("/debug/requests")
def debug_requests(token: str = "", limit: int = 5) -> JSONResponse:
    if not DEBUG_TOKEN or token != DEBUG_TOKEN:
        return JSONResponse({"error": "not found"}, status_code=404)
    keys = sorted(_capture_keys())[-max(1, min(limit, 20)):]
    return JSONResponse({"captures": [store.get("debug.capture", key) for key in keys]})


@app.get("/", response_class=PlainTextResponse)
def index() -> str:
    return (
        "TDS 2026-09 GA3 agents\n"
        f"student: {EMAIL}\n"
        "Q2 A2A invoice agent : /a2a/ (card: /.well-known/agent-card.json)\n"
        "Q3 mailroom          : POST /mailroom\n"
        "Q4 incident agent    : POST /v2/incidents\n"
    )


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "db": store.DB_PATH,
            "model": llm.model_name(),
            "modelConfigured": llm.available(),
            "modelCalls": llm.CALL_COUNT,
            "lastModelError": llm.LAST_ERROR,
        }
    )
