"""TDS 2026-09 GA3 — one service answering Q2, Q3 and Q4.

  Q2  A2A invoice agent        base  <BASE>/a2a/   (card at <ORIGIN>/.well-known/agent-card.json)
  Q3  Safe AI mailroom         URL   <BASE>/mailroom
  Q4  Observable incident      base  <BASE>        (routes under /v2/incidents)

Run locally:  uvicorn main:app --port 8000
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

from ga3 import incident, invoice, mailroom
from ga3 import llm, store

EMAIL = "24f2001638@ds.study.iitm.ac.in"

app = FastAPI(title="TDS GA3 agents", version=os.environ.get("GA3_VERSION", "1.0.0"))

app.include_router(invoice.router)
app.include_router(mailroom.router)
app.include_router(incident.router)


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
